//! Minimal, dependency-free liveness/readiness HTTP probes.
//!
//! Serves `GET /livez` (always `200 OK` once the listener is bound) and
//! `GET /readyz` (`200 OK` iff the node has drained replication bootstrap, has a
//! populated routing table, and has storage open; `503` otherwise) on a
//! loopback [`tokio::net::TcpListener`]. Uses only `tokio` (already a
//! dependency) — no `hyper`, `axum`, or `prometheus`. The server is
//! intentionally minimal: it binds `127.0.0.1`, accepts only `GET`, caps
//! concurrent connections, applies a read timeout, and closes each connection
//! after one response (no keep-alive).
//!
//! Wire this into the node by calling [`serve`] with a [`ReadinessProbe`] built
//! from the running node's handles (see `RunningNode::run`).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use saorsa_core::P2PNode;

use crate::logging::{debug, error, info, warn};

/// Maximum concurrent probe connections. Generous for orchestrator / load-
/// balancer probes (typically one connection); caps file-descriptor and task
/// usage against a probe flood.
const MAX_HEALTH_CONNECTIONS: usize = 16;

/// Per-connection read timeout (slow-loris resistance).
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Readiness inputs, captured as cheap cloneable handles so the probe task can
/// answer without borrowing the (`&mut self`-consuming) node run loop.
#[derive(Clone)]
pub struct ReadinessProbe {
    /// Replication bootstrap flag (`true` while bootstrapping). `None` when the
    /// node runs with replication / storage disabled — such a node has no
    /// bootstrap phase to drain and is treated as drained.
    is_bootstrapping: Option<Arc<RwLock<bool>>>,
    /// Handle to the [`P2PNode`], for routing-table size queries.
    p2p_node: Arc<P2PNode>,
    /// Minimum routing-table peers required for readiness (close-group size).
    min_routing_peers: usize,
    /// Whether chunk storage is open / enabled on this node.
    storage_open: bool,
}

impl ReadinessProbe {
    /// Build a probe from the running node's handles.
    ///
    /// `is_bootstrapping` is the replication engine's bootstrap flag (or `None`
    /// when replication is disabled); `min_routing_peers` is the close-group
    /// size; `storage_open` is whether chunk storage is enabled.
    #[must_use]
    pub fn new(
        is_bootstrapping: Option<Arc<RwLock<bool>>>,
        p2p_node: Arc<P2PNode>,
        min_routing_peers: usize,
        storage_open: bool,
    ) -> Self {
        Self {
            is_bootstrapping,
            p2p_node,
            min_routing_peers,
            storage_open,
        }
    }

    /// Evaluate current readiness (see [`readyz_ready`]).
    ///
    /// Reads the *live* bootstrap flag and routing-table size, so a node that
    /// loses its peers after a partition correctly drops back to "not ready"
    /// until it recovers a full close group.
    pub async fn is_ready(&self) -> bool {
        let bootstrap_drained = match &self.is_bootstrapping {
            Some(flag) => !*flag.read().await,
            None => true,
        };
        let routing_peers = self.p2p_node.dht_manager().get_routing_table_size().await;
        readyz_ready(
            bootstrap_drained,
            routing_peers,
            self.min_routing_peers,
            self.storage_open,
        )
    }
}

/// Pure readiness predicate for `/readyz`.
///
/// A node is ready iff it has drained replication bootstrap
/// (`bootstrap_drained`), currently holds at least `min_routing_peers`
/// routing-table peers (a full close group is needed to serve a store/get
/// against any address), and has storage open (`storage_open`).
#[must_use]
pub const fn readyz_ready(
    bootstrap_drained: bool,
    routing_peers: usize,
    min_routing_peers: usize,
    storage_open: bool,
) -> bool {
    bootstrap_drained && routing_peers >= min_routing_peers && storage_open
}

/// Run the liveness/readiness probe server until `shutdown` fires.
///
/// Binds `127.0.0.1:{port}`. A `port` of `0` disables the server (returns
/// immediately). Never returns an error: a bind failure is logged and the task
/// exits, because the probes are best-effort operability, not a node
/// dependency.
pub async fn serve(port: u16, probe: ReadinessProbe, shutdown: CancellationToken) {
    if port == 0 {
        debug!("Health probe server disabled (port 0)");
        return;
    }
    let addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind health probe server to {addr}: {e}");
            return;
        }
    };
    info!("Health probe server listening on http://{addr}/livez and /readyz");

    let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_HEALTH_CONNECTIONS));
    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                debug!("Health probe server shutting down");
                return;
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((mut socket, _peer)) => {
                        let probe = probe.clone();
                        let sem = Arc::clone(&semaphore);
                        tokio::spawn(async move {
                            let Ok(_permit) = sem.try_acquire() else {
                                warn!(
                                    "Health probe server at connection limit, dropping connection"
                                );
                                return;
                            };
                            handle_connection(&mut socket, &probe).await;
                        });
                    }
                    Err(e) => warn!("Failed to accept health probe connection: {e}"),
                }
            }
        }
    }
}

/// Read one request, route `/livez` / `/readyz`, write one response, close.
async fn handle_connection(socket: &mut tokio::net::TcpStream, probe: &ReadinessProbe) {
    use tokio::io::AsyncReadExt;

    let mut buf = [0u8; 1024];
    // IO error or read timeout -> drop the connection.
    let Ok(Ok(n)) = tokio::time::timeout(READ_TIMEOUT, socket.read(&mut buf)).await else {
        return;
    };
    if n == 0 {
        return;
    }
    let request = String::from_utf8_lossy(&buf[..n]);

    // `GET`-only; reject other methods with `405`.
    if !request.starts_with("GET ") {
        let _ = write_response(socket, "405 Method Not Allowed", "method not allowed\n").await;
        return;
    }
    // Reject path traversal / NUL bytes.
    if request.contains("..") || request.contains('\0') {
        let _ = write_response(socket, "400 Bad Request", "bad request\n").await;
        return;
    }

    // Parse the request target precisely (`GET <target> HTTP/1.1`) and strip any
    // query string, so `/livezfoo` does not match `/livez`.
    let target = request
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/");

    let _ = match target {
        "/livez" => write_response(socket, "200 OK", "ok\n").await,
        "/readyz" => {
            if probe.is_ready().await {
                write_response(socket, "200 OK", "ready\n").await
            } else {
                write_response(socket, "503 Service Unavailable", "not ready\n").await
            }
        }
        _ => {
            write_response(
                socket,
                "404 Not Found",
                "not found; try /livez or /readyz\n",
            )
            .await
        }
    };
}

/// Write a minimal `Connection: close` `HTTP/1.1` response with a computed
/// `Content-Length`.
async fn write_response(
    socket: &mut tokio::net::TcpStream,
    status: &str,
    body: &str,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let response = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len(),
    );
    socket.write_all(response.as_bytes()).await
}

#[cfg(test)]
mod tests {
    use super::readyz_ready;

    /// Representative close-group size for the predicate tests.
    const K: usize = 8;

    #[test]
    fn readyz_requires_bootstrap_peers_and_storage() {
        // Not bootstrap-drained -> never ready, regardless of peers / storage.
        assert!(!readyz_ready(false, K, K, true));
        assert!(!readyz_ready(false, K + 100, K, true));

        // Drained but below the peer threshold -> not ready.
        assert!(!readyz_ready(true, 0, K, true));
        assert!(!readyz_ready(true, K - 1, K, true));

        // Drained + enough peers but storage closed -> not ready.
        assert!(!readyz_ready(true, K, K, false));

        // Drained + >= threshold peers + storage open -> ready.
        assert!(readyz_ready(true, K, K, true));
        assert!(readyz_ready(true, K + 1, K, true));
    }
}

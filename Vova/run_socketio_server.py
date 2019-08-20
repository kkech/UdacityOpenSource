import torch
import syft as sy
from socketio_server import WebsocketIOServerWorker

# Use Numpy serialization strategy
sy.serde._serialize_tensor = sy.serde.numpy_tensor_serializer
sy.serde._deserialize_tensor = sy.serde.numpy_tensor_deserializer
sy.serde._apply_compress_scheme = sy.serde.apply_no_compression

hook = sy.TorchHook(torch)

server_worker = WebsocketIOServerWorker(hook, "localhost", 5000, log_msgs=True, 
	cors_allowed_origins=["https://vvmnnnkv.github.io", "https://localhost:8080", "http://localhost:8080"])
app = server_worker.app

if __name__ == "__main__":
    server_worker.start()

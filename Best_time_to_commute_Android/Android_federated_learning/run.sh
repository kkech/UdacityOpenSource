pip list | grep -E 'torch|syft|torchvision'
echo "Starting server #1 - port 5001"
nohup python3 -u ./socketio_server1.py >> output.log &
echo "Starting server #1 - port 5002"
nohup python3 -u ./socketio_server2.py >> output.log &
echo "Starting server #1 - port 5003"
nohup python3 -u ./socketio_server3.py >> output.log &
echo "Starting server #1 - port 5004"
nohup python3 -u ./socketio_server4.py >> output.log &
echo "Waiting 10 seconds..."
sleep 10
netstat -tulpn | grep python3
echo "Starting notebook"
jupyter notebook Main.ipynb

class SyftConnectorSocketIO extends SyftConnector {

    constructor(url) {
        super();
        this.url = url
        this.socket = io(this.url)
        this.socket.on('connect', () => this.dispatchEvent(new CustomEvent('connect')))
        this.socket.on('message', (data) => this.dispatchEvent(new CustomEvent('message', {detail: data})))
        this.socket.on('disconnect', () => this.dispatchEvent(new CustomEvent('disconnect')))
        this.socket.on('connect_error', (err) => this.dispatchEvent(new CustomEvent('error', {detail: err})))
    }

    ack() {
        this.socket.emit('client_ack', '1')
    }

    send(message) {
        this.socket.emit('client_send_result', message)
    }
}

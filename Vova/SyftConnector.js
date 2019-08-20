class SyftConnector extends EventTarget {

    constructor() {
        super()
    }

    ack() {

    }

    send() {

    }

    onConnect(listener) {
        this.addEventListener('connect', listener)
    }

    onMessage(listener) {
        this.addEventListener('message', listener)
    }

    onDisconnect(listener) {
        this.addEventListener('disconnect', listener)
    }

    onError(listener) {
        this.addEventListener('error', listener)
    }
}

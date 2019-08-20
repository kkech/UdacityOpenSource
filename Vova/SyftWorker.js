SYFT_COMMAND_MAP = new Map([
    ['__add__', COMMAND_ADD],
    ['__radd__', COMMAND_ADD],
    ['__mul__', COMMAND_MUL],
    ['__matmul__', COMMAND_MATMUL],
    ['__truediv__', COMMAND_DIV],
])

class SyftWorker extends EventTarget {

    constructor(connector, backend, logger) {
        super()
        this.connector = connector
        this.backend = backend
        this.logger = logger || { debug: () => {}, log: () => {}}
        this.connector.onMessage((e) => {
            this.logger.debug("Message received")
            this.handleMessage(e.detail)
        })
        this.connector.onConnect((e) => {
            this.logger.debug("Connected")
            this.dispatchEvent(new CustomEvent('connect'))
        })
        this.connector.onError((e) => {
            this.logger.debug("Connection error", e.detail)
        })
    }

    onReady(listener) {
        this.addEventListener('connect', listener)
    }

    handleMessage(message) {
        const syMessage = SyftMessage.fromBinary(message)
        this.logger.debug("Message parsed:", syMessage)
        switch (syMessage.msg_type) {
            case MSGTYPE_OBJ:
                this.addObject(syMessage.contents)
                this.connector.ack()
                break

            case MSGTYPE_OBJ_DEL:
            case MSGTYPE_FORCE_OBJ_DEL:
                const delTensorId = syMessage.contents
                this.deleteTensor(delTensorId)
                this.connector.ack()
                break

            case MSGTYPE_CMD:
                const command = syMessage.contents
                this.executeCommand(command)
                this.connector.ack()
                break

            case MSGTYPE_OBJ_REQ:
                const getTensorId = syMessage.contents
                const tensorOut = this.getTensor(getTensorId)
                const response = new SyftMessage(null, tensorOut)
                this.connector.send(response.toBinary())
                break

            case MSGTYPE_IS_NONE:
                this.connector.send(
                    new SyftMessage(
                        null, 
                        !this.backend.tensorExist(syMessage.contents.id)
                    ).toBinary()
                )
                break

            default:
                throw new Error(`Unsupported message type: ${syMessage.msg_type}`)
        }
    }

    addObject(obj) {
        if (obj instanceof SyftTorchTensor) {
            this.addTensor(obj.id, obj)
        } else {
            throw new Error(`Unsupported object type: ${typeof obj}`)
        }
    }

    addTensor(tensorId, tensor) {
        this.logger.debug(`Adding tensor ${tensorId}`)
        this.backend.addTensor(tensorId, tensor)
    }

    getTensor(tensorId) {
        this.logger.debug(`Retrieving tensor ${tensorId}`)
        if (!this.backend.tensorExist(tensorId)) {
            throw Error(`Tensor is missing: ${tensorId}`)
        }
        let tensor = this.backend.getTensor(tensorId)
        return tensor
    }

    deleteTensor(tensorId) {
        this.logger.debug(`Removing tensor ${tensorId}`)
        this.backend.deleteTensor(tensorId)
    }

    executeCommand(command) {
        let [[cmd, arg1, [arg2], unk], [res_id]] = command
        this.logger.debug(`Executing command: ${cmd}`, arg1, arg2, unk, `Result id: ${res_id}`)

        if (!(arg1 instanceof SyftPointerTensor)) {
            throw new Error(`Arg1 is not supported type`)
        }
        if (!this.backend.tensorExist(arg1.id_at_location)) {
            throw new Error(`Tensor not found ${arg1.id_at_location}`)
        }
        let arg2Type
        if (typeof arg2 === "number") {
            // do nothing
            arg2Type = "number"
        } else if (arg2 instanceof SyftPointerTensor) {
            if (!this.backend.tensorExist(arg2.id_at_location)) {
                throw new Error(`Tensor from arg2 not found: ${arg2.id_at_location}`)
            }
            arg2 = arg2.id_at_location
            arg2Type = "tensorId"
        } else {
            throw new Error(`Arg2 has unsupported type`)
        }

        const backendCmd = SYFT_COMMAND_MAP.get(cmd)
        if (!backendCmd) {
            throw new Error(`Unsupported command ${cmd}`)
        }
        this.backend.compute(backendCmd, arg1.id_at_location, arg2Type, arg2, res_id)
    }
}

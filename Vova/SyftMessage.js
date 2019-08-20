const MSGTYPE_CMD = 1
const MSGTYPE_OBJ = 2
const MSGTYPE_OBJ_REQ = 3
const MSGTYPE_OBJ_DEL = 4
const MSGTYPE_EXCEPTION = 5
const MSGTYPE_IS_NONE = 6
const MSGTYPE_GET_SHAPE = 7
const MSGTYPE_SEARCH = 8
const MSGTYPE_FORCE_OBJ_DEL = 9

class SyftMessage {

    constructor(msg_type, contents) {
        this.msg_type = msg_type
        this.contents = contents
    }

    static fromBinary(blob) {
        let data = new Uint8Array(blob)
        switch (data[0]) {
            case COMPRESSION_NONE:
                data = data.subarray(1)
                break
            default:
                throw new SyftMessageParseError(`Unsupported compression ${data[0]}`)
                break
        }

        let rawData = MessagePack.decode(data)
        return detail(null, rawData)
    }

    toBinary() {
        let payload
        if (this.msg_type) {
            payload = simplify(PyTuple.from([this.msg_type, this.contents]))
        } else {
            payload = simplify(this.contents)
        }

        let encoded = MessagePack.encode(payload)

        // add compression
        let result = new Uint8Array(encoded.length + 1)
        result.set(Uint8Array.of(COMPRESSION_NONE), 0)
        result.set(encoded, 1)
        return result.buffer
    }

    static detail(worker, obj) {
        const [msg_type, contents_ser] = obj
        const contents = detail(worker, contents_ser)
        switch (msg_type) {
//            case MSGTYPE_CMD:
//                return new SyftOperation(contents[0], contents[1])
//                break
            default:
                return new SyftMessage(msg_type, contents)
                break
        }
    }

}

class SyftOperation extends SyftMessage {

    constructor(message, return_ids) {
        this.msg_type = MSGTYPE_CMD
        this.message = message
        this.return_ids = return_ids
    }

}
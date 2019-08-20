const COMMAND_ADD = 'add'
const COMMAND_MUL = 'mul'
const COMMAND_MATMUL = 'matmul'
const COMMAND_DIV = 'div'

class SyftBackend extends EventTarget {
    tensors = {}

    toTensor(value) {
        // no op
        return value
    }

    toSyftTensor(tensorId, tensor) {
        // no op
        return tensor
    }

    addTensor(tensorId, tensor) {
        this.tensors[tensorId] = this.toTensor(tensor)
        this.dispatchEvent(new CustomEvent('tensor-added', {detail: tensorId}))
    }

    getTensor(tensorId) {
        const tensor = this.tensors[tensorId]
        return this.toSyftTensor(tensorId, tensor)
    }

    deleteTensor(tensorId) {
        delete this.tensors[tensorId]
        this.dispatchEvent(new CustomEvent('tensor-removed', {detail: tensorId}))
    }

    tensorExist(tensorId) {
        return tensorId in this.tensors
    }

    compute(command, tensorId, argType, arg, resultId) {
        const tensor = this.tensors[tensorId]
        let operand
        switch (argType) {
            case 'number':
                operand = arg
                break
            case 'tensorId':
                operand = this.tensors[arg]
                break
        }
        const result = this.executeOp(command, tensor, operand)
        this.dispatchEvent(new CustomEvent('tensor-op', {detail: {op: command, tensorId: tensorId, arg: arg}}))
        this.addTensor(resultId, result)
    }

}

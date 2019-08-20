const TF_COMMAND_MAP = new Map([
    [COMMAND_ADD, 'add'],
    [COMMAND_MUL, 'mul'],
    [COMMAND_MATMUL, 'matMul'],
    [COMMAND_DIV, 'div'],
])

class SyftBackendTensorFlow extends SyftBackend {

    /**
     * Convert to TF tensor
     * @param value
     * @returns {tf.Tensor}
     */
    toTensor(value) {
        if (value instanceof SyftTorchTensor) {
            return tf.tensor(value.data, value.shape)
        } else if (typeof value === "number") {
            return tf.tensor(value)
        } else if (value instanceof tf.Tensor) {
            return value
        } else {
            throw new Error(`Failed to make tensor from value ${value}`)
        }
    }

    toSyftTensor(tensorId, tensor) {
        const data = tensor.dataSync()
        const shape = tensor.shape
        return new SyftTorchTensor(tensorId, data, shape)
    }

    executeOp(command, tensor, operand) {
        const tfOp = TF_COMMAND_MAP.get(command)
        return tensor[tfOp](operand)
    }

}


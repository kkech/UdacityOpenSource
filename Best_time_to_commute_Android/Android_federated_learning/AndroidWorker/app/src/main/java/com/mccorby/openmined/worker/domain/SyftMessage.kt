package com.mccorby.openmined.worker.domain

sealed class SyftMessage {

    data class ExecuteCommand(val command: SyftCommand) : SyftMessage() // And a list of tensors?

    data class SetObject(val objectToSet: SyftOperand) : SyftMessage() {

        override fun equals(other: Any?): Boolean {
            return objectToSet.id == (other as SetObject).objectToSet.id
        }

        override fun hashCode(): Int {
            return objectToSet.hashCode()
        }
    }

    data class RespondToObjectRequest(val objectToSend: SyftOperand.SyftTensor) : SyftMessage()

    data class DeleteObject(val objectToDelete: Long) : SyftMessage()

    data class GetObject(val tensorPointerId: SyftTensorId) : SyftMessage()

    object OperationAck : SyftMessage() {
        override fun toString(): String {
            return "ACK"
        }
    }
}

sealed class SyftCommand {
    data class Add(val tensors: List<SyftOperand>, val resultIds: List<SyftTensorId>) : SyftCommand()
    data class Multiply(val tensors: List<SyftOperand>, val resultIds: List<SyftTensorId>) : SyftCommand()
}
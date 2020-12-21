package com.mccorby.openmined.worker.domain

sealed class SyftResult {
    class ObjectAdded(val syftObject: SyftOperand) : SyftResult() {
        override fun equals(other: Any?): Boolean {
            return syftObject.id == (other as ObjectAdded).syftObject.id
        }

        override fun hashCode(): Int {
            return syftObject.hashCode()
        }
    }

    class ObjectRemoved(val pointer: SyftTensorId) : SyftResult() {
        override fun equals(other: Any?): Boolean {
            return pointer == (other as ObjectRemoved).pointer
        }

        override fun hashCode(): Int {
            return pointer.hashCode()
        }
    }
    class ObjectRetrieved(val syftObject: SyftOperand.SyftTensor) : SyftResult()
    class CommandResult(val command: SyftCommand, val commandResult: SyftOperand.SyftTensor) : SyftResult()
    object UnexpectedResult : SyftResult()
}
package com.mccorby.openmined.worker.domain

const val NO_ID: SyftTensorId = -1

typealias SyftTensorId = Long

sealed class SyftOperand(open val id: SyftTensorId) {

    data class SyftTensor(override val id: SyftTensorId, val byteArray: ByteArray = ByteArray(0)) : SyftOperand(id) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as SyftTensor

            if (id != other.id) return false
            if (!byteArray.contentEquals(other.byteArray)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = id.hashCode()
            result = 31 * result + byteArray.contentHashCode()
            return result
        }
    }

    data class SyftTensorPointer(override val id: SyftTensorId) : SyftOperand(id)
}

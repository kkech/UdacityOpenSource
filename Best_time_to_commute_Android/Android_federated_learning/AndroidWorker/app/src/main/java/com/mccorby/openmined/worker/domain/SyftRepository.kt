package com.mccorby.openmined.worker.domain

import io.reactivex.Flowable
import java.lang.IllegalArgumentException

// The repository allow us to use different types of data sources without requiring modifying the upper layers
class SyftRepository(private val syftDataSource: SyftDataSource) {

    private val tensorMap = mutableMapOf<Long, SyftOperand.SyftTensor>()

    fun connect() {
        syftDataSource.connect()
    }

    fun onStatusChange(): Flowable<String> = syftDataSource.onStatusChanged()

    fun sendMessage(syftMessage: SyftMessage) {
        // TODO This should be done by the use case
        when (syftMessage) {
            is SyftMessage.OperationAck -> syftDataSource.sendOperationAck(syftMessage)
            is SyftMessage.RespondToObjectRequest -> syftDataSource.sendMessage(syftMessage)
        }
    }

    fun onNewMessage(): Flowable<SyftMessage> = syftDataSource.onNewMessage()

    fun disconnect() {
        syftDataSource.disconnect()
    }

    fun setObject(objectToSet: SyftOperand.SyftTensor) {
        // Create id for this tensor
        tensorMap[objectToSet.id] = objectToSet
    }

    fun setObject(tensorId: SyftTensorId, objectToSet: SyftOperand.SyftTensor) {
        // Create id for this tensor
        tensorMap[tensorId] = objectToSet
    }

    fun getObject(tensorId: SyftTensorId): SyftOperand.SyftTensor {
        val tensor = tensorMap[tensorId]
        return tensor ?: throw IllegalArgumentException("Tensor $tensorId not found")
    }

    fun removeObject(objectToDelete: Long) {
        tensorMap.remove(objectToDelete)
    }
}
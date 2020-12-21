package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult

class DeleteObjectUseCase(private val syftRepository: SyftRepository) {

    operator fun invoke(newSyftMessage: SyftMessage.DeleteObject): SyftResult.ObjectRemoved {
        syftRepository.removeObject(newSyftMessage.objectToDelete)
        syftRepository.sendMessage(SyftMessage.OperationAck)
        return SyftResult.ObjectRemoved(newSyftMessage.objectToDelete)
    }
}
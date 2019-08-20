package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult

class GetObjectUseCase(private val syftRepository: SyftRepository) {

    operator fun invoke(syftMessage: SyftMessage.GetObject): SyftResult.ObjectRetrieved {
        val tensor = syftRepository.getObject(syftMessage.tensorPointerId)
        // TODO copy should not be necessary. Here set just to make it work. This is a value that should have been already set before
        val tensorCopy = tensor.copy(id = syftMessage.tensorPointerId)
        syftRepository.sendMessage(SyftMessage.RespondToObjectRequest(tensorCopy))
        return SyftResult.ObjectRetrieved(tensorCopy)
    }
}
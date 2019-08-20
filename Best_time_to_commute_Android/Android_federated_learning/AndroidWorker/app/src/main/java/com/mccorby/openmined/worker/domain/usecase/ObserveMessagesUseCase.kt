package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import io.reactivex.Flowable

class ObserveMessagesUseCase(
    private val syftRepository: SyftRepository,
    private val setObjectUseCase: SetObjectUseCase,
    private val executeCommandUseCase: ExecuteCommandUseCase,
    private val getObjectUseCase: GetObjectUseCase,
    private val deleteObjectUseCase: DeleteObjectUseCase
) {

    operator fun invoke(): Flowable<SyftResult> {
        return syftRepository.onNewMessage()
            .map { processNewMessage(it) }
    }

    private fun processNewMessage(newSyftMessage: SyftMessage): SyftResult {
        return when (newSyftMessage) {
            is SyftMessage.SetObject -> {
                setObjectUseCase(newSyftMessage)
            }
            is SyftMessage.ExecuteCommand -> {
                executeCommandUseCase(newSyftMessage)
            }
            is SyftMessage.GetObject -> {
                getObjectUseCase(newSyftMessage)
            }
            is SyftMessage.DeleteObject -> {
                deleteObjectUseCase(newSyftMessage)
            }
            else -> {
                SyftResult.UnexpectedResult
            }
        }
    }
}
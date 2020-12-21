package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftRepository
import io.reactivex.Completable

class ConnectUseCase(private val syftRepository: SyftRepository) {

    fun execute(): Completable {
        return Completable.fromAction { syftRepository.connect() }
    }
}
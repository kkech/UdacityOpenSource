package com.mccorby.openmined.worker.ui

import android.arch.lifecycle.ViewModel
import android.arch.lifecycle.ViewModelProvider
import androidx.work.WorkManager
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.usecase.ConnectUseCase
import com.mccorby.openmined.worker.domain.usecase.ObserveMessagesUseCase

class MainViewModelFactory(
    private val observeMessagesUseCase: ObserveMessagesUseCase,
    private val connectUseCase: ConnectUseCase,
    private val syftRepository: SyftRepository,
    private val workManager: WorkManager
) : ViewModelProvider.NewInstanceFactory() {

    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return MainViewModel(observeMessagesUseCase, connectUseCase, syftRepository, workManager) as T
    }
}
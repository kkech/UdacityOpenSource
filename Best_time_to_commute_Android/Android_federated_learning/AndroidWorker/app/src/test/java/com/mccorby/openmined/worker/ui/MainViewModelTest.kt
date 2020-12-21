package com.mccorby.openmined.worker.ui

import androidx.work.WorkManager
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.usecase.ConnectUseCase
import com.mccorby.openmined.worker.domain.usecase.ObserveMessagesUseCase
import io.mockk.mockk
import org.junit.Before

class MainViewModelTest {

    private lateinit var cut: MainViewModel
    private val observeMessagesUseCase = mockk<ObserveMessagesUseCase>(relaxed = true)
    private val connectUseCase = mockk<ConnectUseCase>(relaxed = true)
    private val syftRepository = mockk<SyftRepository>(relaxed = true)
    private val workManager = mockk<WorkManager>(relaxed = true)

    @Before
    fun setUp() {
        cut = MainViewModelFactory(observeMessagesUseCase, connectUseCase, syftRepository, workManager).create(MainViewModel::class.java)
    }
}

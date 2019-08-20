package com.mccorby.openmined.worker.ui

import android.arch.lifecycle.MutableLiveData
import android.arch.lifecycle.ViewModel
import android.util.Log
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import com.mccorby.openmined.worker.domain.SyftCommand
import com.mccorby.openmined.worker.domain.SyftOperand
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import com.mccorby.openmined.worker.domain.usecase.ConnectUseCase
import com.mccorby.openmined.worker.domain.usecase.ObserveMessagesUseCase
import com.mccorby.openmined.worker.services.DisconnectWorkManager
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.schedulers.Schedulers

class MainViewModel(
    private val observeMessagesUseCase: ObserveMessagesUseCase,
    private val connectUseCase: ConnectUseCase,
    private val syftRepository: SyftRepository,
    private val workManager: WorkManager
) : ViewModel() {

    val syftMessageState = MutableLiveData<SyftResult>()
    val syftTensorState = MutableLiveData<SyftOperand.SyftTensor>()
    val viewState = MutableLiveData<String>()

    private val compositeDisposable = CompositeDisposable()

    fun initiateCommunication() {
        val connectDisposable = connectUseCase.execute()
            .subscribeOn(Schedulers.io())
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe()
        compositeDisposable.add(connectDisposable)

        // TODO Ideally we should start listening to message when connectUseCase completes
        startListeningToMessages()
    }

    private fun startListeningToMessages() {
        val messageDisposable = observeMessagesUseCase()
            .map { processNewMessage(it) }
            .subscribeOn(Schedulers.io())
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe()

        val statusDisposable = syftRepository.onStatusChange()
            .map { viewState.postValue(it) }
            .subscribeOn(Schedulers.io())
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe()

        compositeDisposable.addAll(messageDisposable, statusDisposable)
    }

    private fun processNewMessage(syftResult: SyftResult) {
        Log.d("ConnectActivity", "Received new SyftMessage at $syftResult")
        when (syftResult) {
            is SyftResult.ObjectAdded -> {
                syftTensorState.postValue(syftResult.syftObject as SyftOperand.SyftTensor)
            }
            is SyftResult.CommandResult -> {
                processCommand(syftResult)
            }
            is SyftResult.ObjectRetrieved -> {
                viewState.postValue("Server requested tensor with id ${syftResult.syftObject.id}")
            }
            is SyftResult.ObjectRemoved -> {
                viewState.postValue("Tensor with id ${syftResult.pointer} deleted")
            }
            else -> {
                syftMessageState.postValue(syftResult)
            }
        }
    }

    private fun processCommand(commandResult: SyftResult.CommandResult) {
        when (commandResult.command) {
            is SyftCommand.Add -> {
                viewState.postValue("Result of Add:")
                syftTensorState.postValue(commandResult.commandResult)
            }
        }
    }

    public override fun onCleared() {
        compositeDisposable.clear()
        workManager.enqueue(OneTimeWorkRequest.from(DisconnectWorkManager::class.java))
        super.onCleared()
    }
}

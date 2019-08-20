package com.mccorby.openmined.worker.domain

import io.reactivex.Flowable

interface SyftDataSource {
    fun connect()
    fun disconnect()
    fun sendOperationAck(syftMessage: SyftMessage)
    fun sendMessage(syftMessage: SyftMessage)
    fun onNewMessage(): Flowable<SyftMessage>
    fun onStatusChanged(): Flowable<String>
}
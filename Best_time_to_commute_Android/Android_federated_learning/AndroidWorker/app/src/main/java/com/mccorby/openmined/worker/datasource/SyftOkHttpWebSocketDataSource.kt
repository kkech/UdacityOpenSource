package com.mccorby.openmined.worker.datasource

import android.util.Log
import com.mccorby.openmined.worker.datasource.mapper.mapToString
import com.mccorby.openmined.worker.datasource.mapper.mapToSyftMessage
import com.mccorby.openmined.worker.domain.SyftDataSource
import com.mccorby.openmined.worker.domain.SyftMessage
import io.reactivex.Flowable
import io.reactivex.processors.PublishProcessor
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString

private const val TAG = "SyftOkHttpWebSocketDS"

class SyftOkHttpWebSocketDataSource(private val webSocketUrl: String) : SyftDataSource {

    private val okHttpClient = OkHttpClient()
    private val publishProcessor: PublishProcessor<SyftMessage> = PublishProcessor.create<SyftMessage>()
    private val statusPublishProcessor: PublishProcessor<String> = PublishProcessor.create<String>()

    private lateinit var ws: WebSocket

    override fun connect() {
        val request = Request.Builder().url(webSocketUrl).build()
        val listener = object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                // TODO This would be better if it was offering a domain sealed class
                statusPublishProcessor.offer("Connected")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                statusPublishProcessor.offer("Disconnected!")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "Received message from the other side as String")
                val syftMessage = text.toByteArray().mapToSyftMessage()

                Log.d(TAG, "SyftTensor $syftMessage")

                publishProcessor.offer(syftMessage)
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                Log.d(TAG, "Received message from the other side as bytes")
                onMessage(webSocket, bytes.hex())
            }
        }
        ws = okHttpClient.newWebSocket(request, listener)
    }

    override fun disconnect() {
    }

    override fun sendOperationAck(syftMessage: SyftMessage) {
        ws.send(syftMessage.mapToString())
    }

    override fun sendMessage(syftMessage: SyftMessage) {
    }

    override fun onNewMessage(): Flowable<SyftMessage> {
        return publishProcessor.onBackpressureBuffer()
    }

    override fun onStatusChanged(): Flowable<String> {
        return statusPublishProcessor.onBackpressureBuffer()
    }
}
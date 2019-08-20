package com.mccorby.openmined.worker.datasource

import android.util.Log
import com.mccorby.openmined.worker.datasource.mapper.CompressionConstants.NO_COMPRESSION
import com.mccorby.openmined.worker.datasource.mapper.mapToByteArray
import com.mccorby.openmined.worker.datasource.mapper.mapToString
import com.mccorby.openmined.worker.datasource.mapper.mapToSyftMessage
import com.mccorby.openmined.worker.domain.SyftDataSource
import com.mccorby.openmined.worker.domain.SyftMessage
import io.reactivex.Flowable
import io.reactivex.processors.PublishProcessor
import io.socket.client.IO
import io.socket.client.Socket

// This client finished the operation and sends an ACK to the server
private const val SEND_OPERATION_ACK = "client_ack"
// This client sends the result of a request, i.e., when the server requests x_ptr.get()
private const val SEND_RESULT = "client_send_result"
private const val SEND_CLIENT_ID = "client_id"

private const val TAG = "SyftWebSocketDataSource"

class SyftWebSocketDataSource(private val webSocketUrl: String, private val clientId: String) : SyftDataSource {
    private lateinit var socket: Socket
    private val publishProcessor: PublishProcessor<SyftMessage> = PublishProcessor.create<SyftMessage>()
    private val statusPublishProcessor: PublishProcessor<String> = PublishProcessor.create<String>()

    override fun connect() {
        val opts = IO.Options()
        opts.forceNew = true
        socket = IO.socket(webSocketUrl, opts)

        socket.on(Socket.EVENT_MESSAGE) { args -> onEventMessage(args) }

        socket.on(Socket.EVENT_CONNECT) { onConnect() }
        socket.on(Socket.EVENT_DISCONNECT) { onDisconnect() }
        socket.on(Socket.EVENT_CONNECT_ERROR) { args -> Log.d(TAG, "EVENT Connect error ${logError(args)}") }
        socket.on(Socket.EVENT_CONNECT_TIMEOUT) { Log.d(TAG, "EVENT Connect timeout error") }
        socket.on(Socket.EVENT_RECONNECTING) { Log.d(TAG, "EVENT Reconnecting") }
        socket.on(Socket.EVENT_ERROR) { args -> Log.d(TAG, "EVENT Error ${logError(args)}") }
        socket.on(Socket.EVENT_PING) { Log.d(TAG, "EVENT Ping") }
        socket.on(Socket.EVENT_PONG) { Log.d(TAG, "EVENT Pong") }

        socket.connect()
    }

    private fun logError(args: Array<Any>) {
        args.forEach { print(" $it") }
    }

    override fun onStatusChanged(): Flowable<String> = statusPublishProcessor.onBackpressureBuffer()

    override fun disconnect() {
        Log.d(TAG, "Disconnecting")
        socket.disconnect()
    }

    override fun sendOperationAck(syftMessage: SyftMessage) {
        // Simplify, Serialize, and Compress
        // TODO Add mapper from SyftMessage2ByteArray?.
        Log.d(TAG, "Sending message $syftMessage")
        socket.emit(SEND_OPERATION_ACK, syftMessage.mapToString())
    }

    override fun sendMessage(syftMessage: SyftMessage) {
        val byteArray = syftMessage.mapToByteArray()
        // TODO Compression must be added in the mapper!!
        socket.emit(SEND_RESULT, byteArrayOf(NO_COMPRESSION.toByte()) + byteArray)
    }

    override fun onNewMessage(): Flowable<SyftMessage> = publishProcessor.onBackpressureBuffer()

    private fun onConnect() {
        Log.d(TAG, "Connection done")
        statusPublishProcessor.offer("Connected!")
        // Sending a dummy client id. This should be provided
        socket.emit(SEND_CLIENT_ID, clientId)
    }

    private fun onDisconnect() {
        Log.d(TAG, "We're disconnected")
        statusPublishProcessor.offer("Disconnected!")
    }

    private fun onEventMessage(vararg args: Any) {
        // Decompress, Deserialise, Build object
        Log.d(TAG, "Received message from the other side")
        val syftMessage = ((args[0] as Array<Any>)[0] as ByteArray).mapToSyftMessage()

        Log.d(TAG, "SyftTensor $syftMessage")

        publishProcessor.offer(syftMessage)
    }
}

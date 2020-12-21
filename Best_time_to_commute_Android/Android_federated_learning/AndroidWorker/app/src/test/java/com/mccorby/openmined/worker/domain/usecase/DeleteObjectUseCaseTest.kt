package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import io.mockk.Runs
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.verifyOrder
import org.junit.Assert
import org.junit.Before
import org.junit.Test

class DeleteObjectUseCaseTest {

    private val repository = mockk<SyftRepository>(relaxed = true)
    private lateinit var cut: DeleteObjectUseCase

    @Before
    fun setUp() {
        cut = DeleteObjectUseCase(repository)
    }

    @Test
    fun `Given a delete object message then the use case removes it from the repository and send ACK`() {
        val newMessage = mockk<SyftMessage.DeleteObject>()
        val tensorId = 1L
        val expected = SyftResult.ObjectRemoved(tensorId)

        every { newMessage.objectToDelete } returns tensorId
        every { repository.removeObject(any()) } just Runs
        every { repository.sendMessage(any()) } just Runs

        val result = cut(newMessage)

        Assert.assertEquals(expected, result)

        verifyOrder {
            repository.removeObject(tensorId)
            repository.sendMessage(SyftMessage.OperationAck)
        }
    }
}
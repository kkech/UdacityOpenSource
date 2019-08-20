package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftOperand
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import io.mockk.Runs
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.verifyOrder
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class SetObjectUseCaseTest {

    private val repository = mockk<SyftRepository>(relaxed = true)
    private lateinit var cut: SetObjectUseCase

    @Before
    fun setUp() {
        cut = SetObjectUseCase(repository)
    }

    @Test
    fun `Given a setObject message arrives then the object is stored and an ACK is sent back`() {
        val newMessage = mockk<SyftMessage.SetObject>()
        val objectToSet = mockk<SyftOperand.SyftTensor>(relaxed = true)
        val expected = SyftResult.ObjectAdded(objectToSet)

        every { newMessage.objectToSet } returns objectToSet
        every { repository.setObject(any()) } just Runs
        every { repository.sendMessage(any()) } just Runs

        val result = cut(newMessage)

        assertEquals(expected, result)

        verifyOrder {
            repository.setObject(objectToSet)
            repository.sendMessage(SyftMessage.OperationAck)
        }
    }
}
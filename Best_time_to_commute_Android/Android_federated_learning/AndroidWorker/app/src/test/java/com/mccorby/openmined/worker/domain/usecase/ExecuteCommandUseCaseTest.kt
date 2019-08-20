package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.MLFramework
import com.mccorby.openmined.worker.domain.SyftCommand
import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftOperand
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class ExecuteCommandUseCaseTest {

    private lateinit var cut: ExecuteCommandUseCase

    private val syftRepository = mockk<SyftRepository>(relaxed = true)
    private val mlFramework = mockk<MLFramework>()

    @Before
    fun setUp() {
        cut = ExecuteCommandUseCase(syftRepository, mlFramework)
    }

    @Test
    fun `Given an Add command with two tensors then the use case returns the addition of both operands`() {
        val tensor1 = mockk<SyftOperand.SyftTensor>()
        val tensor2 = mockk<SyftOperand.SyftTensor>()
        val addResult = mockk<SyftOperand.SyftTensor>()
        val operands = listOf(tensor1, tensor2)
        val resultId = 1L
        val resultIds = listOf(resultId)
        val addCommand = SyftCommand.Add(operands, resultIds)
        val syftMessage = SyftMessage.ExecuteCommand(addCommand)

        val expected = SyftResult.CommandResult(addCommand, addResult)

        every { mlFramework.add(tensor1, tensor2) } returns addResult

        val result = cut(syftMessage)

        verify {
            mlFramework.add(tensor1, tensor2)
            syftRepository.setObject(resultId, addResult)
            syftRepository.sendMessage(SyftMessage.OperationAck)
        }

        assertEquals(expected.command, result.command)
        assertEquals(expected.commandResult, result.commandResult)
    }

    @Test
    fun `Given an Add command with two tensor pointers then the use case returns the addition of both operands`() {
        val tensorPointer1 = SyftOperand.SyftTensorPointer(1)
        val tensorPointer2 = SyftOperand.SyftTensorPointer(2)
        val tensor1 = mockk<SyftOperand.SyftTensor>()
        val tensor2 = mockk<SyftOperand.SyftTensor>()
        val addResult = mockk<SyftOperand.SyftTensor>()
        val operands = listOf(tensorPointer1, tensorPointer2)
        val resultId = 3L
        val resultIds = listOf(resultId)
        val addCommand = SyftCommand.Add(operands, resultIds)
        val syftMessage = SyftMessage.ExecuteCommand(addCommand)

        val expected = SyftResult.CommandResult(addCommand, addResult)

        every { syftRepository.getObject(tensorPointer1.id) } returns tensor1
        every { syftRepository.getObject(tensorPointer2.id) } returns tensor2
        every { mlFramework.add(tensor1, tensor2) } returns addResult

        val result = cut(syftMessage)

        verify {
            syftRepository.getObject(tensorPointer1.id)
            syftRepository.getObject(tensorPointer2.id)
            mlFramework.add(tensor1, tensor2)
            syftRepository.setObject(resultId, addResult)
            syftRepository.sendMessage(SyftMessage.OperationAck)
        }

        assertEquals(expected.command, result.command)
        assertEquals(expected.commandResult, result.commandResult)
    }
}
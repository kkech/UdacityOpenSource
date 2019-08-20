package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftOperand
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import io.mockk.every
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class GetObjectUseCaseTest {

    private val repository = mockk<SyftRepository>(relaxed = true)
    private lateinit var cut: GetObjectUseCase

    @Before
    fun setUp() {
        cut = GetObjectUseCase(repository)
    }

    @Test
    fun `Given a get Object message then the use case returns the item in the repository`() {
        val tensorId = 1L
        val syftMessage = SyftMessage.GetObject(tensorId)
        val tensor = mockk<SyftOperand.SyftTensor>(relaxed = true)
        val expected = SyftResult.ObjectRetrieved(tensor)

        every { repository.getObject(tensorId) } returns tensor
        every { tensor.id } returns tensorId
        every { tensor.copy(id = tensorId) } returns tensor

        val result = cut(syftMessage)

        assertEquals(expected.syftObject, result.syftObject)
    }
}
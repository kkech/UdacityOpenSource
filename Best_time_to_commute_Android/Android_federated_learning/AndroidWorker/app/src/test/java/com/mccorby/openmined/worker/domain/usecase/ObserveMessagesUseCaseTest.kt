package com.mccorby.openmined.worker.domain.usecase

import com.mccorby.openmined.worker.domain.SyftCommand
import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftOperand
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import io.mockk.every
import io.mockk.mockk
import io.mockk.verifyOrder
import io.reactivex.Flowable
import org.junit.Before
import org.junit.Test

class ObserveMessagesUseCaseTest {

    private val repository = mockk<SyftRepository>()
    private val setObjectUseCase = mockk<SetObjectUseCase>(relaxed = true)
    private val executeCommandUseCase = mockk<ExecuteCommandUseCase>(relaxed = true)
    private val getObjectUseCase = mockk<GetObjectUseCase>(relaxed = true)
    private val deleteObjectUseCase = mockk<DeleteObjectUseCase>(relaxed = true)

    private lateinit var cut: ObserveMessagesUseCase

    @Before
    fun setUp() {
        cut = ObserveMessagesUseCase(repository, setObjectUseCase, executeCommandUseCase, getObjectUseCase, deleteObjectUseCase)
    }

    @Test
    fun `Given a SetObject message then use case forwards it to SetObject use case`() {
        val newMessage = mockk<SyftMessage.SetObject>(relaxed = true)
        val objectToSet = mockk<SyftOperand>(relaxed = true)
        val expected = SyftResult.ObjectAdded(objectToSet)

        every { setObjectUseCase(newMessage) } returns expected
        every { repository.onNewMessage() } returns Flowable.just(newMessage)

        val testObserver = cut().test()

        testObserver.assertNoErrors()
            .assertValue(expected)

        verifyOrder {
            setObjectUseCase(newMessage)
        }
    }

    @Test
    fun `Given an execute command message arrives then the object is retrieved and Response is sent back`() {
        val newMessage = mockk<SyftMessage.ExecuteCommand>(relaxed = true)
        val syftCommand = mockk<SyftCommand>(relaxed = true)
        val commandResult = mockk<SyftOperand.SyftTensor>(relaxed = true)
        val expected = SyftResult.CommandResult(syftCommand, commandResult)

        every { executeCommandUseCase(newMessage) } returns expected
        every { repository.onNewMessage() } returns Flowable.just(newMessage)

        val testObserver = cut().test()

        testObserver.assertNoErrors()
            .assertValue(expected)

        verifyOrder {
            executeCommandUseCase(newMessage)
        }
    }

    @Test
    fun `Given a getObject message then the use case retrieves the object and returns it in a response`() {
        val tensorId = 1L
        val newMessage = SyftMessage.GetObject(tensorId)
        val tensor = mockk<SyftOperand.SyftTensor>()
        val expected = SyftResult.ObjectRetrieved(tensor)

        every { repository.onNewMessage() } returns Flowable.just(newMessage)
        every { getObjectUseCase(newMessage) } returns expected

        val testObserver = cut().test()

        testObserver.assertNoErrors()
            .assertValue(expected)

        verifyOrder {
            getObjectUseCase(newMessage)
        }
    }

    @Test
    fun `Given a deleteObject message then the use case deletes the object and returns ACK`() {
        val tensorId = 1L
        val newMessage = SyftMessage.DeleteObject(tensorId)
        val expected = SyftResult.ObjectRemoved(tensorId)

        every { repository.onNewMessage() } returns Flowable.just(newMessage)
        every { deleteObjectUseCase(newMessage) } returns expected

        val testObserver = cut().test()

        testObserver.assertNoErrors()
            .assertValue(expected)

        verifyOrder {
            deleteObjectUseCase(newMessage)
        }
    }
}
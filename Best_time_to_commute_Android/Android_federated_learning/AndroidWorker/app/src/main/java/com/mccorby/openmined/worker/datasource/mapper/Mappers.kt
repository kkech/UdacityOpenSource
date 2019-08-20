package com.mccorby.openmined.worker.datasource.mapper

import com.mccorby.openmined.worker.datasource.mapper.CommandConstants.CMD_ADD
import com.mccorby.openmined.worker.datasource.mapper.CommandConstants.CMD_MULTIPLY
import com.mccorby.openmined.worker.datasource.mapper.CompressionConstants.COMPRESSION_ENABLED
import com.mccorby.openmined.worker.datasource.mapper.OperationConstants.CMD
import com.mccorby.openmined.worker.datasource.mapper.OperationConstants.FORCE_OBJ_DEL
import com.mccorby.openmined.worker.datasource.mapper.OperationConstants.OBJ
import com.mccorby.openmined.worker.datasource.mapper.OperationConstants.OBJ_DEL
import com.mccorby.openmined.worker.datasource.mapper.OperationConstants.OBJ_REQ
import com.mccorby.openmined.worker.datasource.mapper.TypeConstants.TYPE_TENSOR
import com.mccorby.openmined.worker.datasource.mapper.TypeConstants.TYPE_TENSOR_POINTER
import com.mccorby.openmined.worker.domain.SyftCommand
import com.mccorby.openmined.worker.domain.SyftMessage
import com.mccorby.openmined.worker.domain.SyftOperand
import org.msgpack.core.MessagePack
import org.msgpack.value.ArrayValue
import org.msgpack.value.Value
import org.msgpack.value.impl.ImmutableArrayValueImpl
import org.msgpack.value.impl.ImmutableLongValueImpl
import org.msgpack.value.impl.ImmutableNilValueImpl
import org.msgpack.value.impl.ImmutableStringValueImpl

private const val TAG = "MapperDS"

// These are values defined in PySyft
internal object CompressionConstants {
    const val COMPRESSION_ENABLED = 49
    const val NO_COMPRESSION = 40
}

// Operations in PySyft
internal object OperationConstants {
    internal const val CMD = 1
    internal const val OBJ = 2
    internal const val OBJ_REQ = 3
    internal const val OBJ_DEL = 4
    internal const val EXCEPTION = 5
    internal const val IS_NONE = 6
    internal const val GET_SHAPE = 7
    internal const val SEARCH = 8
    internal const val FORCE_OBJ_DEL = 9
}

// Types are encoded in the stream sent from PySyft
internal object TypeConstants {
    const val TYPE_TUPLE = 2
    const val TYPE_LIST = 3
    const val TYPE_TENSOR = 12
    const val TYPE_TENSOR_POINTER = 18
}

// Commands
internal object CommandConstants {
    const val CMD_ADD = "__add__"
    const val CMD_MULTIPLY = "__mul__"
}

// TODO Probably Json or something similar. The name should reflect the format
fun SyftMessage.mapToString(): String {
    val packer = MessagePack.newDefaultBufferPacker()
    return when (this) {
        is SyftMessage.OperationAck -> packer.packString(SyftMessage.OperationAck.toString())
        else -> {
            packer
        }
    }.toString()
}

fun SyftMessage.mapToByteArray(): ByteArray {
    // Assuming we are sending just a tensor
    // (0, (91189711850, b"numpy stuff", None, None, None, None))
    val packer = MessagePack.newDefaultBufferPacker()

    val operationArray = ImmutableArrayValueImpl(
        arrayOf<Value>(
            ImmutableLongValueImpl(TYPE_TENSOR.toLong()),
            ImmutableArrayValueImpl(
                arrayOf<Value>(
                    ImmutableLongValueImpl((this as SyftMessage.RespondToObjectRequest).objectToSend.id),
                    ImmutableStringValueImpl(objectToSend.byteArray),
                    // TODO Fill these values!
                    ImmutableNilValueImpl.get(),
                    ImmutableNilValueImpl.get(),
                    ImmutableNilValueImpl.get(),
                    ImmutableNilValueImpl.get()
                )
            )
        )
    )

    packer.packValue(operationArray)
    return packer.toByteArray()
}

fun ByteArray.mapToSyftMessage(): SyftMessage {
    // (tensor.id, tensor_bin, chain, grad_chain, tags, tensor.description)
    // Remove first byte indicating if stream has been compressed or not
    val isCompress = this[0]
    val byteArray = if (isCompress.toInt() == COMPRESSION_ENABLED) {
        decompress(this.drop(1).toByteArray())
    } else {
        this.drop(1).toByteArray()
    }

    val unpacker = MessagePack.newDefaultUnpacker(byteArray)
    val streamToDecode = unpacker.unpackValue()
    val operationDto = unpackOperation(streamToDecode.asArrayValue()[1].asArrayValue())

    return mapOperation(operationDto)
}

private fun unpackOperation(operationArray: ArrayValue): OperationDto {
    val operation = operationArray[0].asIntegerValue().asInt()
    val operands = operationArray[1]
    return when (operation) {
        OBJ -> unpackObjectSet(operands)
        CMD -> unpackCommand(operands)
        OBJ_DEL, FORCE_OBJ_DEL -> unpackObjectDelete(operands)
        OBJ_REQ -> unpackObjectRequest(operands)
        else -> {
            TODO("Operation $operation not yet implemented! $operationArray")
        }
    }
}

private fun unpackObjectSet(operands: Value): OperationDto {
    val data = unpackOperandByType(operands as ArrayValue)
    return OperationDto(OBJ, "", listOf(data))
}

private fun unpackObjectDelete(operands: Value): OperationDto {
    val operand = operands.asNumberValue().toLong()
    val pointerDto = OperandDto.TensorPointerDto()
    pointerDto.id = operand
    return OperationDto(OBJ_DEL, value = listOf((pointerDto)))
}

fun unpackObjectRequest(operands: Value): OperationDto {
    val operand = operands.asNumberValue().toLong()
    val pointerDto = OperandDto.TensorPointerDto()
    pointerDto.id = operand
    return OperationDto(OBJ_REQ, value = listOf((pointerDto)))
}

fun unpackCommand(operands: Value): OperationDto {
    // At this point we should have a list with the form [2, [command, [first_operand][list of other operands]][return_ids]

    // [2,[[2,["__add__",[11,[9999,6830]],[2,[[11,[9999,1234]]]]]],[3,[7766]]]]
    val operationComponents = operands.asArrayValue()[1].asArrayValue()
    val operation =
        operationComponents[0].asArrayValue()[1].asArrayValue() // ["__add__",[11,[9999,6830]],[2,[[11,[9999,1234]]]]]
    val returnIds = operationComponents[1].asArrayValue() // [3, [7766]]

    return when (val command = unpackCommand(operation)) { // [18,["__add__"]]
        CMD_ADD -> {
            val operationDto = OperationDto(op = CMD, command = command)
            val tensorList = mutableListOf<OperandDto>()
            val op1 = operation[1].asArrayValue()
            val op2 = operation[2].asArrayValue()[1]
            tensorList.add(unpackOperandByType(op1))

            op2.asArrayValue().map {
                val operand = unpackOperandByType(it.asArrayValue())
                tensorList.add(operand)
            }

            operationDto.value = tensorList.toList()
            operationDto.returnId = returnIds[1].asArrayValue().map { it.asNumberValue().toLong() }
            operationDto
        }
        CMD_MULTIPLY -> {
            val operationDto = OperationDto(op = CMD, command = command)
            val tensorList = mutableListOf<OperandDto>()
            val op1 = operation[1].asArrayValue()
            val op2 = operation[2].asArrayValue()[1]
            tensorList.add(unpackOperandByType(op1))

            op2.asArrayValue().map {
                val operand = unpackOperandByType(it.asArrayValue())
                tensorList.add(operand)
            }

            operationDto.value = tensorList.toList()
            operationDto.returnId = returnIds[1].asArrayValue().map { it.asNumberValue().toLong() }
            operationDto
        }
        else -> {
            TODO("$command not yet implemented!")
        }
    }
}

// [18,["__add__"]]
private fun unpackCommand(operation: ArrayValue) =
    operation[0].asArrayValue()[1].asArrayValue()[0].asStringValue().asString()

private fun unpackOperandByType(streamToDecode: ArrayValue): OperandDto {
    val type = streamToDecode[0].asIntegerValue().toInt()
    val operandArray = streamToDecode.drop(1)[0].asArrayValue()
    return when (type) {
        TYPE_TENSOR -> mapTensor(operandArray)
        TYPE_TENSOR_POINTER -> mapTensorPointer(operandArray)
        else -> {
            TODO("$type not yet implemented")
        }
    }
}

private fun mapTensorPointer(streamToDecode: ArrayValue): OperandDto.TensorPointerDto {
    // (64458802353, 7201727941, 'bob', None, torch.Size([1, 2]))
    val tensorDto = OperandDto.TensorPointerDto()
    tensorDto.id = streamToDecode[1].asNumberValue().toLong()
    // TODO The rest of attributes will come later
    return tensorDto
}

private fun mapTensor(streamToDecode: ArrayValue): OperandDto {
    val tensorDto = OperandDto.TensorDto()
    tensorDto.id = streamToDecode[0].asNumberValue().toLong()
    tensorDto.data = streamToDecode[1].asStringValue().asByteArray()
    // 3 -> chain
    // 4 -> grad_chain
    // 5 -> tags
    // 6 -> tensor description
    return tensorDto
}

fun decompress(stream: ByteArray): ByteArray {
    TODO("LZ4 Compression Not yet Implemented")
//    val factory = LZ4Factory.fastestInstance()
//    // Size is not known. It could be sent in the tuple
//    val decompressor = factory.safeDecompressor()
//    var dest = ByteArray(8096)
//    val decompressedLength = decompressor.decompress(stream, dest)
//    return dest
}

private fun mapOperation(operationDto: OperationDto): SyftMessage {
    return when (operationDto.op) {
        OBJ -> {
            val operand = mapOperandToDomain(operationDto.value.first())
            SyftMessage.SetObject(operand)
        }
        CMD -> {
            val listOfSyftOperands = operationDto.value.map {
                mapOperandToDomain(it)
            }
            val command = createCommandMessage(operationDto.command, listOfSyftOperands, operationDto.returnId)
            SyftMessage.ExecuteCommand(command)
        }
        OBJ_DEL, FORCE_OBJ_DEL -> {
            SyftMessage.DeleteObject((operationDto.value[0] as OperandDto.TensorPointerDto).id)
        }
        OBJ_REQ -> {
            SyftMessage.GetObject((operationDto.value[0] as OperandDto.TensorPointerDto).id)
        }
        else -> {
            throw IllegalArgumentException("Operation ${operationDto.op} not yet supported")
        }
    }
}

private fun createCommandMessage(command: String, listOfSyftOperands: List<SyftOperand>, returnId: List<Long>): SyftCommand {
    return when (command) {
        CMD_ADD -> {
            SyftCommand.Add(listOfSyftOperands, returnId)
        }
        CMD_MULTIPLY -> {
            SyftCommand.Multiply(listOfSyftOperands, returnId)
        }
        else -> {
            TODO("Command $command not yet implemented")
        }
    }
}

private fun mapOperandToDomain(dto: OperandDto): SyftOperand {
    return when (dto) {
        is OperandDto.TensorDto -> SyftOperand.SyftTensor(dto.id, dto.data)
        is OperandDto.TensorPointerDto -> SyftOperand.SyftTensorPointer(dto.id)
    }
}

class OperationDto(
    var op: Int = 0,
    var command: String = "",
    var value: List<OperandDto> = mutableListOf(),
    var returnId: List<Long> = listOf()
) {
    override fun toString(): String {
        return "$op - $command - $value"
    }
}

sealed class OperandDto {
    class TensorDto : OperandDto() {
        var id: Long = 0
        var data: ByteArray = byteArrayOf()
    }

    class TensorPointerDto : OperandDto() {
        var id: Long = 0
    }
}

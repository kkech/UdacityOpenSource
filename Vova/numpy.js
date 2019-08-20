// From https://gist.github.com/nvictus/88b3b5bfe587d32ac1ab519fd0009607
function asciiDecode(buf) {
    return String.fromCharCode.apply(null, new Uint8Array(buf));
}

function readUint16LE(buffer) {
    var view = new DataView(buffer);
    var val = view.getUint8(0);
    val |= view.getUint8(1) << 8;
    return val;
}

function fromArrayBuffer(buf, offsetPos) {
    // Check the magic number
    let pos = offsetPos
    if (asciiDecode(buf.slice(pos + 1, pos + 6)) != 'NUMPY') {
        throw new Error('unknown file type');
    }

    var version = new Uint8Array(buf.slice(pos + 6, pos + 8)),
        headerLength = readUint16LE(buf.slice(pos + 8, pos + 10)),
        headerStr = asciiDecode(buf.slice(pos + 10, pos + 10 + headerLength));
    offsetBytes = pos + 10 + headerLength;

    // Hacky conversion of dict literal string to JS Object
    eval("var info = " + headerStr.toLowerCase().replace('(', '[').replace('),', ']'));

    // Intepret the bytes according to the specified dtype
    let numEl = info.shape.reduce((a, b) => a * b, 1);
    var data;
    if (info.descr === "|u1") {
        data = new Uint8Array(buf, offsetBytes, numEl);
    } else if (info.descr === "|i1") {
        data = new Int8Array(buf, offsetBytes, numEl);
    } else if (info.descr === "<u2") {
        data = new Uint16Array(buf, offsetBytes, numEl);
    } else if (info.descr === "<i2") {
        data = new Int16Array(buf, offsetBytes, numEl);
    } else if (info.descr === "<u4") {
        data = new Uint32Array(buf, offsetBytes, numEl);
    } else if (info.descr === "<i4") {
        data = new Int32Array(buf, offsetBytes, numEl);
    } else if (info.descr === "<f4") {
        data = new Float32Array(buf, offsetBytes, numEl);
    } else if (info.descr === "<f8") {
        data = new Float64Array(buf, offsetBytes, numEl);
    } else {
        throw new Error('unknown numeric dtype')
    }

    return {
        shape: info.shape,
        fortran_order: info.fortran_order,
        data: data
    };
}

// From https://github.com/propelml/tfjs-npy/blob/master/npy.ts
function toNumpyBuffer(data, shape) {
    const dtype = data.constructor.name
    let descr, bytesPerElement
    switch (dtype) {
        case 'Float32Array':
            descr = '<f4'
            bytesPerElement = 4
            break
        case 'Int32Array':
            descr = '<i4'
            bytesPerElement = 4
            break
        default:
            throw Error(`Data type ${dtype} is not supported`)
    }

    // First figure out how long the file is going to be so we can create the
    // output ArrayBuffer.
    const magicStr = "NUMPY";
    const versionStr = "\x01\x00";
    const shapeStr = String(shape.join(",")) + ",";
    const [d, fo, s] = [descr, "False", shapeStr];
    let header = `{'descr': '${d}', 'fortran_order': ${fo}, 'shape': (${s}), }`;
    const unpaddedLength =
        1 + magicStr.length + versionStr.length + 2 + header.length;
    // Spaces to 16-bit align.
    const padding = " ".repeat((16 - unpaddedLength % 16) % 16);
    header += padding;
    const numEls = shape.reduce((a, b) => a * b, 1)
    const dataLen = bytesPerElement * numEls;
    const totalSize = unpaddedLength + padding.length + dataLen;

    const ab = new ArrayBuffer(totalSize);
    const view = new DataView(ab);
    let pos = 0;

    // Write magic string and version.
    view.setUint8(pos++, 0x93);
    pos = writeStrToDataView(view, magicStr + versionStr, pos);

    // Write header length and header.
    view.setUint16(pos, header.length, true);
    pos += 2;
    pos = writeStrToDataView(view, header, pos);

    // Write data
    for (let i = 0; i < data.length; i++) {
        switch (dtype) {
            case "Float32Array":
                view.setFloat32(pos, data[i], true);
                pos += 4;
                break;

            case "Int32Array":
                view.setInt32(pos, data[i], true);
                pos += 4;
                break;

            default:
                throw Error(`dtype ${tensor.dtype} not yet supported.`);
        }
    }
    return ab;
}

function writeStrToDataView(view, str, pos) {
    for (let i = 0; i < str.length; i++) {
        view.setInt8(pos + i, str.charCodeAt(i));
    }
    return pos + str.length;
}

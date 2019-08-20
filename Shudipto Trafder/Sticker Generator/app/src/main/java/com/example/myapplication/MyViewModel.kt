package com.example.myapplication

import android.util.Base64
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.example.myapplication.rest.WebService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream

class MyViewModel:ViewModel(){

    private val job = SupervisorJob()
    protected val uiScope = CoroutineScope(Dispatchers.Main + job)

    val state = MutableLiveData<Status>()

    fun request(id: String) {

        Timber.i("Request")

        uiScope.launch(Dispatchers.IO) {
            val service = WebService.getService()

            try{
                val pojo = service.getImages(id)
                val output = pojo?.output ?: ""

                // if api key did not match then it will return a string which is less than 30
                if (output.length >= 30 ){
                    state.postValue(Status(1, tempFile(output)))
                } else{
                    val s = Status(0, "Something went wrong. Try again?")
                    state.postValue(s)
                }
            } catch (e:Exception){
                val s = Status(0, "Something went wrong. Try again?")
                state.postValue(s)
            }

        }

    }

    private fun tempFile(output: String):String{
        val file = File.createTempFile("output", ".arr")
        val encoder = Base64.decode(output, Base64.DEFAULT)
        Timber.i("Byte Array Length: ${encoder.size}")
        val stream = FileOutputStream(file)
        stream.write(encoder)
        stream.flush()
        stream.close()
        return file.absolutePath
    }


}




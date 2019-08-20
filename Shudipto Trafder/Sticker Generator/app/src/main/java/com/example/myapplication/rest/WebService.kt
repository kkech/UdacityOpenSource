package com.example.myapplication.rest

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class WebService{
    companion object{
        var webservice:WebInterface ?= null

        @Synchronized
        fun getService():WebInterface {
            if (webservice == null){
                val retrofit = Retrofit.Builder()
                    .baseUrl("https://anime-generator.herokuapp.com/")
                    .addConverterFactory(GsonConverterFactory.create())
                    .build()

                webservice = retrofit.create(WebInterface::class.java)
            }

            return webservice!!
        }
    }
}
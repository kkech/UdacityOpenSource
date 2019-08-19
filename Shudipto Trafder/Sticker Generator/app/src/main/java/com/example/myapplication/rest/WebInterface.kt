package com.example.myapplication.rest

import retrofit2.http.GET
import retrofit2.http.Path

interface WebInterface {
    @GET("01732033963/{num}")
    suspend fun getImages(@Path("num") number: String): POJO?
}
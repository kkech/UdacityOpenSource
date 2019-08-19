package com.example.myapplication.ext

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity

inline fun <reified T : AppCompatActivity> AppCompatActivity.next(finish: Boolean = false) {
    startActivity(Intent(this, T::class.java))
    if (finish) finish()
}
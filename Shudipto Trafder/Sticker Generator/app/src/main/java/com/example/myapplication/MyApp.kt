package com.example.myapplication

import android.app.Application
import com.example.myapplication.ext.NetworkLiveData
import com.rohitss.uceh.UCEHandler
import timber.log.Timber

class MyApp:Application(){

    override fun onCreate() {
        super.onCreate()

        // add network live data
        NetworkLiveData.init(this)

        UCEHandler.Builder(this).build()

        if (BuildConfig.DEBUG){
            Timber.plant(Timber.DebugTree())
        }

    }

}


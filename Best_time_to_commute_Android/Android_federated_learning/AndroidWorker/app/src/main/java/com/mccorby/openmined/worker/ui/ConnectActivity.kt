package com.mccorby.openmined.worker.ui

import android.arch.lifecycle.Observer
import android.arch.lifecycle.ViewModelProviders
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import androidx.work.WorkManager
import com.mccorby.openmined.worker.R
import com.mccorby.openmined.worker.datasource.SyftWebSocketDataSource
import com.mccorby.openmined.worker.domain.SyftOperand
import com.mccorby.openmined.worker.domain.SyftRepository
import com.mccorby.openmined.worker.domain.SyftResult
import com.mccorby.openmined.worker.domain.usecase.ConnectUseCase
import com.mccorby.openmined.worker.domain.usecase.DeleteObjectUseCase
import com.mccorby.openmined.worker.domain.usecase.ExecuteCommandUseCase
import com.mccorby.openmined.worker.domain.usecase.GetObjectUseCase
import com.mccorby.openmined.worker.domain.usecase.ObserveMessagesUseCase
import com.mccorby.openmined.worker.domain.usecase.SetObjectUseCase
import com.mccorby.openmined.worker.framework.DL4JFramework
import com.mccorby.openmined.worker.framework.toINDArray
import kotlinx.android.synthetic.main.activity_main.*

class ConnectActivity : AppCompatActivity() {

    private lateinit var viewModel: MainViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btn_initiate.setOnClickListener { viewModel.initiateCommunication() }

        injectDependencies()
    }

    // TODO Inject using Kodein or another DI framework
    private fun injectDependencies() {
        val clientId = "Android-${System.currentTimeMillis()}"
        val webSocketUrl = "http://10.0.2.2:5003"
        val syftDataSource = SyftWebSocketDataSource(webSocketUrl, clientId)
        val syftRepository = SyftRepository(syftDataSource)
        val mlFramework = DL4JFramework()
        val setObjectUseCase = SetObjectUseCase(syftRepository)
        val executeCommandUseCase = ExecuteCommandUseCase(syftRepository, mlFramework)
        val getObjectUseCase = GetObjectUseCase(syftRepository)
        val deleteObjectUseCase = DeleteObjectUseCase(syftRepository)
        val observeMessagesUseCase = ObserveMessagesUseCase(
            syftRepository,
            setObjectUseCase,
            executeCommandUseCase,
            getObjectUseCase,
            deleteObjectUseCase
        )
        val connectUseCase = ConnectUseCase(syftRepository)

        viewModel = ViewModelProviders.of(
            this,
            MainViewModelFactory(observeMessagesUseCase, connectUseCase, syftRepository, WorkManager.getInstance())
        ).get(MainViewModel::class.java)

        viewModel.syftMessageState.observe(this, Observer<SyftResult> {
            log_area.append(it.toString() + "\n")
        })
        viewModel.syftTensorState.observe(this, Observer<SyftOperand.SyftTensor> {
            log_area.append(it!!.toINDArray().toString() + "\n")
        })
        viewModel.viewState.observe(this, Observer {
            log_area.append(it + "\n")
        })
    }
}

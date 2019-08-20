package com.example.myapplication

import android.app.ProgressDialog
import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import com.google.android.material.snackbar.Snackbar
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProviders
import com.example.myapplication.ext.NetworkLiveData
import com.example.myapplication.ext.next

import kotlinx.android.synthetic.main.activity_second.*
import kotlinx.android.synthetic.main.content_second.*

@Suppress("DEPRECATION")
class SecondActivity : AppCompatActivity() {

    private val vm:MyViewModel by lazy {
        ViewModelProviders.of(this).get(MyViewModel::class.java)
    }

    var dialog:ProgressDialog ?= null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)
        setSupportActionBar(toolbar)

        dialog = ProgressDialog(this)


        NetworkLiveData.observe(this, Observer {
            if (it && shown){
                button.isClickable = true
                Snackbar.make(secondLay, "Connected", Snackbar.LENGTH_SHORT).show()
                shown = true
            } else if(it==false){
                button.isClickable = false
                Snackbar.make(secondLay, "No internet Available", Snackbar.LENGTH_SHORT).show()
                shown = true
            }
        })

        // detect sate
        vm.state.observe(this, Observer {
            it?.let {
                if (it.id == 1){
                    if (dialog?.isShowing==true) dialog?.dismiss()
                    val intent = Intent(this, FinalActivity::class.java)
                    intent.putExtra("Path", it.message)
                    startActivity(intent)

                } else if (it.id ==0){
                    if (dialog?.isShowing==true) dialog?.dismiss()
                    Snackbar.make(secondLay, it.message, Snackbar.LENGTH_SHORT).show()
                }
            }
        })

        button.setOnClickListener {
            var id = textInputLayout.editText?.text?.toString() ?: "16"
            id = id.trim()

            if (id.isEmpty()) id = "16"

            if (NetworkLiveData.isNetworkAvaiable()) {
                vm.request(id)
                dialog?.setMessage("Getting data from server")
                dialog?.show()
            } else{
                Snackbar.make(secondLay, "No internet Available", Snackbar.LENGTH_SHORT).show()
                shown = true
            }


        }

    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.action_about -> next<AboutActivity>()
        }

        return super.onOptionsItemSelected(item)
    }

    companion object{
        var shown = false
    }


}

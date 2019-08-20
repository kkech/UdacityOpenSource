package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Environment.getExternalStoragePublicDirectory
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import com.google.android.material.snackbar.Snackbar
import kotlinx.android.synthetic.main.activity_final.*
import kotlinx.android.synthetic.main.content_final.*
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*


class FinalActivity : AppCompatActivity() {

    private val code = 121

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_final)
        setSupportActionBar(toolbar)

        val data = intent.getStringExtra("Path")

        if (data == null){
            textView2.text = "Something went wrong! please try again."
        } else{
            process(data)
        }


        fab.setOnClickListener {
            save()
        }

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    private fun save() {
        //check permission first

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED) {
                requestPermissions(arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), code)
            } else{
                savePic()
            }
        } else savePic()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>,
                                            grantResults: IntArray) {

        if (requestCode == code){
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED) {
                    savePic()
                    Timber.i("Permission Granted")
                } else{
                    Snackbar.make(finalLay, "Sorry you have to provide write permission.", Snackbar.LENGTH_SHORT).show()
                    Timber.i("Permission Denied")
                }
            }
        }


        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

    }

    @Suppress("DEPRECATION")
    private fun savePic(){

        val path = getExternalStoragePublicDirectory(
            Environment.DIRECTORY_PICTURES
        )

        Timber.i("File: ${path.path}")

        //due to lack of time use this process
        val date = Date()

        val ext = "${date.day}-${date.hours}-${date.minutes}"

        val name = "Sticker Generator $ext.png"
        val file = File(path.absolutePath, name)
        val stream = FileOutputStream(file)

        Timber.i("File: saved ${file.path}")

        try {
            val bit = imageView3.drawable.toBitmap()
            val pic = bit.compress(Bitmap.CompressFormat.PNG, 100, stream)
            if (pic){
                Snackbar.make(finalLay, "File Saved at ${file.path}", Snackbar.LENGTH_LONG).show()
            } else{
                Snackbar.make(finalLay, "Something went wrong. Try again!", Snackbar.LENGTH_SHORT).show()
            }
        } catch (e:Exception){
            Timber.e(e)
            Snackbar.make(finalLay, "Something went wrong. Try again!", Snackbar.LENGTH_SHORT).show()
        } finally {
            stream.flush()
            stream.close()
        }
    }



    private fun process(path: String) {
        val file = File(path)
        Timber.i("Final path: ${file.path}")
        val arr:ByteArray = file.readBytes()
        Timber.i("Byte Array Length: ${arr.size}")
        val bitmap = BitmapFactory.decodeByteArray(arr, 0, arr.size)
        imageView3.setImageBitmap(bitmap)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            android.R.id.home -> onBackPressed()
        }

        return super.onOptionsItemSelected(item)
    }

}

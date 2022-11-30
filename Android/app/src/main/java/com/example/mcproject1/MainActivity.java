package com.example.mcproject1;

import static android.widget.Toast.LENGTH_SHORT;
import static android.widget.Toast.makeText;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.LruCache;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;

public class MainActivity extends AppCompatActivity {


    private static final int PERMISSION_CODE = 1234;
    private static final int CAPTURE_CODE = 1001;
    MaterialButton captureButton;
    Uri image_uri;
    private static LruCache<String, Bitmap> mMemoryCache;


    @Override
    public void onCreate(Bundle savedInstanceBundle) {
        super .onCreate(savedInstanceBundle);
        setContentView(R.layout.activity_main);

        captureButton = findViewById(R.id.captureButton);

        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED ||
                            checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED ||
                            checkSelfPermission(Manifest.permission.INTERNET) == PackageManager.PERMISSION_DENIED) {

                        String[] permission = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.INTERNET};
                        requestPermissions(permission, PERMISSION_CODE);
                    } else {
                        openCamera();
                    }
                } else {
                    openCamera();
                }
            }
        });


    }

    private void openCamera() {
        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.Images.Media.TITLE, "new image");
        contentValues.put(MediaStore.Images.Media.DESCRIPTION, "From the Camera");
        image_uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, image_uri);
        startActivityForResult(cameraIntent, CAPTURE_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case PERMISSION_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    openCamera();
                } else {
                    makeText(this, "Permission Denied", LENGTH_SHORT).show();
                }
        }
    }

    @SuppressLint("MissingSuperCall")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {

            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(MainActivity.this.getContentResolver().openInputStream(image_uri));
                //bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                LRUCache.getInstance().addBitmapToMemoryCache("image", bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
//            byte[] byteArray = stream.toByteArray();

//            Intent categoryIntent = new Intent(MainActivity.this, SelectCategory.class);
//            categoryIntent.putExtra("imageByteArray", byteArray);

//            categoryIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
//            startActivity(categoryIntent);
            Intent uploadToServerIntent = new Intent(MainActivity.this, Upload.class);
            uploadToServerIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
            startActivity(uploadToServerIntent);
        }
    }
}
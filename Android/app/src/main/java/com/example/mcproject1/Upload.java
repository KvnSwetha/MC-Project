package com.example.mcproject1;

import static com.example.mcproject1.R.string.serverUrlSabharieshUbuntu;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.google.android.material.button.MaterialButton;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class Upload extends Activity {

    MaterialButton saveButton;
    Bitmap bmp;
    Bitmap scaledBmp;
    ByteArrayOutputStream stream;

    @Override
    public void onCreate(Bundle savedInstanceBundle) {
        super.onCreate(savedInstanceBundle);
        setContentView(R.layout.upload);

        bmp = LRUCache.getInstance().getBitmapFromMemCache("image");
        scaledBmp = scaleDown(bmp);
        ImageView image = (ImageView) findViewById(R.id.capturedImageView);
        image.setImageBitmap(bmp);


        saveButton = findViewById(R.id.saveButton);
        saveButton.setEnabled(true);
        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                System.out.println("In save");
                saveButton.setEnabled(false);
                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {

                        stream = new ByteArrayOutputStream();
                        bmp = LRUCache.getInstance().getBitmapFromMemCache("image");
                        bmp.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                        byte[] byteArray = stream.toByteArray();
                        String encodedImage = Base64.encodeToString(byteArray, Base64.DEFAULT);

                        JSONObject json = new JSONObject();
                        try {
                            json.put("image", encodedImage);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                        OkHttpClient client = new OkHttpClient().newBuilder().build();
                        MediaType mediaType = MediaType.parse("application/json");
                        RequestBody body = RequestBody.create(mediaType, json.toString());
                        Request request = new Request.Builder()
                                .url("http://192.168.0.5:8081/upload")
                                .method("POST", body)
                                .addHeader("Content-Type", "application/json")
                                .build();
                        try {
                            Response response = client.newCall(request).execute();
                            System.out.println(response.toString());
                            if(response.isSuccessful()) {
                                showClassification(response.body().string());
                                LRUCache.getInstance().removeBitmapFromMemoryCache("image");
                                Intent mainActivityIntent = new Intent(Upload.this, MainActivity.class);
                                mainActivityIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                                startActivity(mainActivityIntent);
                            } else {
                                System.out.println(response.toString());
                                System.out.println(response.body().toString());
//                                progressBar.setVisibility(View.GONE);
                                showErrorDialog();
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                            System.out.println("In Catch");
                            showErrorDialog();
                        }
                    }
                });
                thread.start();
            }
        });
    }

    public void showErrorDialog() {

        Upload.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                AlertDialog.Builder builder = new AlertDialog.Builder(Upload.this);

                builder.setTitle("Upload Failed");
                builder.setMessage("Error while Uploading Image to Server");
                builder.setNegativeButton("Try Again", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        System.out.println("Try again");
                    }
                });
                builder.setPositiveButton("Go Back Home", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        System.out.println("Go Home");
                        LRUCache.getInstance().removeBitmapFromMemoryCache("image");
                        Intent mainActivityIntent = new Intent(Upload.this, MainActivity.class);
                        mainActivityIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                        startActivity(mainActivityIntent);
                    }
                });
                builder.show();
            }
        });
    }

    public void showClassification(String classification) {

        Upload.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Upload.this, "Image is classified as " + classification, Toast.LENGTH_LONG).show();
            }
        });
    }

    public static Bitmap scaleDown(Bitmap realImage) {
        Bitmap newBitmap = Bitmap.createScaledBitmap(realImage, 375,
                500, true);
        return newBitmap;
    }
}

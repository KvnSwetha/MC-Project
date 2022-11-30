package com.example.mcproject1;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;

import com.google.android.material.button.MaterialButton;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class SelectCategory extends Activity {

    Spinner spinner;
    MaterialButton saveButton;
    Bitmap bmp;
    Bitmap scaledBmp;
    ByteArrayOutputStream stream;

    @Override
    public void onCreate(Bundle savedInstanceBundle) {
        super.onCreate(savedInstanceBundle);
        setContentView(R.layout.select_category);

        bmp = LRUCache.getInstance().getBitmapFromMemCache("image");
        scaledBmp = scaleDown(bmp);
        ImageView image = (ImageView) findViewById(R.id.capturedImageView);
        image.setImageBitmap(bmp);

        spinner = findViewById(R.id.spinner);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.categories, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_item);
        spinner.setAdapter(adapter);

        saveButton = findViewById(R.id.saveButton);
        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                System.out.println("In save");
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
                            json.put("category", spinner.getSelectedItem().toString());
                            json.put("image", encodedImage);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                        OkHttpClient client = new OkHttpClient().newBuilder().build();
                        MediaType mediaType = MediaType.parse("application/json");
                        RequestBody body = RequestBody.create(mediaType, json.toString());
                        Request request = new Request.Builder()
                                .url(getString(R.string.serverUrl) + "upload")
                                .method("POST", body)
                                .addHeader("Content-Type", "application/json")
                                .build();
                        try {
                            Response response = client.newCall(request).execute();
                            System.out.println(response.toString());

                            if(response.isSuccessful()) {
                                LRUCache.getInstance().removeBitmapFromMemoryCache("image");
                                Intent mainActivityIntent = new Intent(SelectCategory.this, MainActivity.class);
                                mainActivityIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                                startActivity(mainActivityIntent);
                            } else {
                                System.out.println(response.toString());
                                System.out.println(response.body().toString());
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

        SelectCategory.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                AlertDialog.Builder builder = new AlertDialog.Builder(SelectCategory.this);

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
                        Intent mainActivityIntent = new Intent(SelectCategory.this, MainActivity.class);
                        mainActivityIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                        startActivity(mainActivityIntent);
                    }
                });
                builder.show();
            }
        });
    }

    public static Bitmap scaleDown(Bitmap realImage) {
        Bitmap newBitmap = Bitmap.createScaledBitmap(realImage, 375,
                500, true);
        return newBitmap;
    }
}

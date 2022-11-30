package com.example.mcproject1;

import android.graphics.Bitmap;
import android.util.LruCache;

public final class LRUCache {

    private static LRUCache cacheInstance;
    private static LruCache<String, Bitmap> mMemoryCache;

    private LRUCache() {
        final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);

        final int cacheSize = maxMemory / 2;

        mMemoryCache = new LruCache<String, Bitmap>(cacheSize) {
            @Override
            protected int sizeOf(String key, Bitmap bitmap) {
                return bitmap.getByteCount() / 1024;
            }
        };
    }

    public static LRUCache getInstance(){
        if(cacheInstance == null)
            cacheInstance = new LRUCache();
        return cacheInstance;
    }

    public void addBitmapToMemoryCache(String key, Bitmap bitmap) {
        if (getBitmapFromMemCache(key) == null) {
            mMemoryCache.put(key, bitmap);
        }
    }

    public Bitmap getBitmapFromMemCache(String key) {
        return mMemoryCache.get(key);
    }

    public void removeBitmapFromMemoryCache(String key) {
        mMemoryCache.remove(key);
    }

}

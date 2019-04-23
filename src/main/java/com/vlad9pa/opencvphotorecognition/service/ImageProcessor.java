package com.vlad9pa.opencvphotorecognition.service;

import org.springframework.web.multipart.MultipartFile;

public interface ImageProcessor {
    byte[] detectFace(MultipartFile file);
}

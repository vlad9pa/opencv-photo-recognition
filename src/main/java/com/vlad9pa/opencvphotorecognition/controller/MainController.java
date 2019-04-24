package com.vlad9pa.opencvphotorecognition.controller;

import com.vlad9pa.opencvphotorecognition.service.ImageProcessor;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequiredArgsConstructor
public class MainController {

    private final ImageProcessor imageProcessor;

    @SneakyThrows
    @PostMapping(value = "/faceDetect/image", produces = MediaType.IMAGE_JPEG_VALUE)
    public byte[] detectFaceImage(@RequestParam("file") MultipartFile file){
       return imageProcessor.detectFace(file);
    }

    @SneakyThrows
    @PostMapping(value = "/eyeDetect/image", produces = MediaType.IMAGE_JPEG_VALUE)
    public byte[] detectEyeImage(@RequestParam("file") MultipartFile file){
        return imageProcessor.detectEye(file);
    }

    @SneakyThrows
    @PostMapping(value = "/white/back", produces = MediaType.IMAGE_JPEG_VALUE)
    public byte[] eleOnWhiteBack(@RequestParam("file") MultipartFile file){
        return imageProcessor.backgroundColor(file);
    }
}

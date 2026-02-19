package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/tracks")
@CrossOrigin(origins = "*")
public class MusicTrackController {

    @Autowired
    private MusicTrackRepository repository;

    private final String PYTHON_SERVICE_URL = "http://127.0.0.1:8000/api/generate";

    @GetMapping("/history")
    public List<MusicTrack> getHistory() {
        return repository.findAllByOrderByCreatedAtDesc();
    }

    @PostMapping("/generate")
    public ResponseEntity<?> generateTrack(@RequestBody Map<String, Object> params) {
        try {
            RestTemplate restTemplate = new RestTemplate();
            Map<String, Object> pyResponse = restTemplate.postForObject(PYTHON_SERVICE_URL, params, Map.class);

            if (pyResponse != null && (Boolean) pyResponse.get("success")) {
                MusicTrack track = MusicTrack.builder()
                        .filename((String) pyResponse.get("filename"))
                        .mood((String) pyResponse.get("mood"))
                        .genre((String) pyResponse.get("genre"))
                        .tempo((Integer) pyResponse.get("tempo"))
                        .style((String) pyResponse.get("style"))
                        .aiEngine((String) pyResponse.get("ai_engine"))
                        .fileSizeKb(Double.valueOf(pyResponse.get("size_kb").toString()))
                        .createdAt(LocalDateTime.now())
                        .build();

                MusicTrack saved = repository.save(track);
                return ResponseEntity.ok(saved);
            }
            return ResponseEntity.status(500).body("Python engine failed to generate music.");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Backend Error: " + e.getMessage());
        }
    }

    @PostMapping("/save")
    public ResponseEntity<MusicTrack> saveTrack(@RequestBody MusicTrack track) {
        track.setCreatedAt(LocalDateTime.now());
        MusicTrack saved = repository.save(track);
        return ResponseEntity.ok(saved);
    }

    @DeleteMapping("/clear")
    public ResponseEntity<?> clearHistory() {
        repository.deleteAll();
        return ResponseEntity.ok(Map.of("success", true));
    }

    @GetMapping("/models")
    public ResponseEntity<?> getAvailableModels() {
        try {
            RestTemplate restTemplate = new RestTemplate();
            Object models = restTemplate.getForObject("http://127.0.0.1:8000/api/models", Object.class);
            return ResponseEntity.ok(models);
        } catch (Exception e) {
            return ResponseEntity.ok(List.of());
        }
    }
}

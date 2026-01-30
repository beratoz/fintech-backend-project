package com.fintech.api.controller;

import com.fintech.api.model.TradeSignal;
import com.fintech.api.service.SignalService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/signals")
public class SignalController {

    private final SignalService signalService;

    @Autowired
    public SignalController(SignalService signalService) {
        this.signalService = signalService;
    }

    @GetMapping
    public List<TradeSignal> getRecentSignals() {
        return signalService.getRecentSignals();
    }

    @GetMapping("/{ticker}")
    public List<TradeSignal> getSignalsByTicker(@PathVariable String ticker) {
        return signalService.getSignalsByTicker(ticker);
    }

    @GetMapping("/latest/{ticker}")
    public TradeSignal getLatestSignal(@PathVariable String ticker) {
        return signalService.getLatestSignal(ticker);
    }
}

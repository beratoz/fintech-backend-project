package com.fintech.api.service;

import com.fintech.api.model.TradeSignal;
import com.fintech.api.repository.SignalRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SignalService {

    private final SignalRepository signalRepository;

    @Autowired
    public SignalService(SignalRepository signalRepository) {
        this.signalRepository = signalRepository;
    }

    public List<TradeSignal> getAllSignals() {
        return signalRepository.findAll();
    }

    public List<TradeSignal> getRecentSignals() {
        return signalRepository.findTop100ByOrderByTimestampDesc();
    }

    public List<TradeSignal> getSignalsByTicker(String ticker) {
        return signalRepository.findByTickerOrderByTimestampDesc(ticker);
    }

    public TradeSignal getLatestSignal(String ticker) {
        return signalRepository.findTopByTickerOrderByTimestampDesc(ticker);
    }
}

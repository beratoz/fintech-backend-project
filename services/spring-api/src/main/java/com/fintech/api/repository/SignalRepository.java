package com.fintech.api.repository;

import com.fintech.api.model.TradeSignal;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SignalRepository extends JpaRepository<TradeSignal, Long> {
    
    // Find all signals for a specific ticker, ordered by time desc
    List<TradeSignal> findByTickerOrderByTimestampDesc(String ticker);
    
    // Get the very latest signal for a ticker
    TradeSignal findTopByTickerOrderByTimestampDesc(String ticker);
    
    // Find recent signals across all tickers
    List<TradeSignal> findTop100ByOrderByTimestampDesc();
}

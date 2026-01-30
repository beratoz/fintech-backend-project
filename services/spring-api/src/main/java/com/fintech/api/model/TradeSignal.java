package com.fintech.api.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Entity
@Table(name = "trade_signals")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class TradeSignal {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String ticker;
    
    // Stored as BigInt (Unix Timestamp MS) in DB
    private Long timestamp;
    
    private Double price;
    
    private Integer prediction; // 1 (BUY) or 0 (HOLD)

    // Storing JSON as String for simplicity in this MVP
    // In a real app, we might use a custom converter or JSONB type
    @Column(columnDefinition = "jsonb")
    private String indicators;
}

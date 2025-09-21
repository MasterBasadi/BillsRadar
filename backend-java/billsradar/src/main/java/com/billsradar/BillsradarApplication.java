package com.billsradar;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BillsradarApplication {

    public static void main(String[] args) {

        SpringApplication.run(BillsradarApplication.class, args);
        System.out.println("CLICK HERE -> http://localhost:8081/");
    }

}

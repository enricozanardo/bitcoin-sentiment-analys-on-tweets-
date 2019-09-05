#Makefile

## Configuration
BUILD_TIME := $(shell date +%FT%T%z)
PROJECT := $(shell basename $(PWD))

## Start the application in development mode
.PHONY: dev
dev:
	cd app

## Install dependencies
.PHONY: install
install:
	pip install -r requirements.txt

// #![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # Tracel

/// Heat SDK
pub mod heat {
    #[cfg(feature = "heat-sdk")]
    pub use heat_sdk::*;
}

# Disable Secure Boot in shim-signed

Open a terminal (Ctrl + Alt + T), and execute sudo mokutil --disable-validation.

Enter a temporary password between 8 to 16 digits. (For example, 12345678, we will use this password later

Enter the same password again to confirm.

Reboot the system and press any key when you see the blue screen (MOK management 

Select Change Secure Boot state

Enter the password you had selected in Step 2 and press Enter. 

Select Yes to disable Secure Boot in shim-signed. 

Press Enter key to finish the whole procedure. 

To re-enable Secure Boot validation in shim, simply run sudo mokutil --enable-validation. 

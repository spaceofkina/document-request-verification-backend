const express = require('express');
const router = express.Router();
const adminController = require('../controllers/adminController');
const { protect } = require('../middleware/auth');
const roleCheck = require('../middleware/roleCheck');

// Admin-only routes
router.use(protect, roleCheck('admin', 'staff'));

// Request management
router.get('/requests', adminController.getAllRequests);
router.get('/requests/:id', adminController.getRequestDetails);
router.put('/requests/:id/status', adminController.updateRequestStatus);
router.post('/requests/:id/verify', adminController.verifyDocument);

// Dashboard
router.get('/stats', adminController.getDashboardStats);

// Document generation
router.get('/requests/:id/preview', adminController.generateDocumentPreview);
router.get('/requests/:id/attachments/:filename', adminController.downloadAttachment);

module.exports = router;